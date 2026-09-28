# Multi-service mTLS deployment example

This example runs one active coordinator and three passive-party services as separate containers. Each passive party receives only its own feature volume. The coordinator receives only Bank features/labels. Services communicate over HTTPS with short-lived demo mTLS certificates plus bearer authentication.

This is a reproducible developer-laptop example, **not a production architecture recommendation**.

## Run

From the repository root:

```bash
mkdir -p deploy/compose/output
docker compose -f deploy/compose/compose.yml up \
  --build --abort-on-container-exit --exit-code-from coordinator
```

The coordinator writes:

```text
deploy/compose/output/compose-smoke.json
```

The record contains the deterministic probability digest, logical message/payload accounting, measured application-layer HTTP bytes, summed HTTP round-trip time, and runtime metadata.

Clean up:

```bash
docker compose -f deploy/compose/compose.yml down -v
```

## Isolation model

`data-init` creates deterministic aligned synthetic data and writes each party block to a different named volume. Runtime services mount only their own volume:

- coordinator: Bank features + labels;
- Telecom service: Telecom features only;
- Insurance service: Insurance features only;
- Retail service: Retail features only.

The data-generation container sees all synthetic blocks only during fixture creation. This is analogous to a test-data provisioning step, not to a real organization boundary.

## Transport controls exercised

- HTTPS;
- mutual TLS with a short-lived demo CA;
- service-specific server certificates;
- Bank client certificate;
- bearer authentication;
- server-side allowed-sender checks;
- bounded request/response implementation from `RemoteHTTPTransport` / `RemotePartyService`;
- replay/idempotency/rate-limit controls provided by the reference transport implementation;
- compressed NumPy serialization (`npy-zlib-base64`);
- TLS handshake health checks;
- deterministic CPU-only smoke training;
- logical and measured application-HTTP communication metadata.

## Security boundary

mTLS authenticates/encrypts the network channel. It does not make legitimate VFL residuals, logits, gradients, model parameters, timing, or other protocol messages information-free. See `docs/threat_model.md`, `docs/privacy_boundaries.md`, and `docs/limitations.md`.

For production deployment, add organization-managed PKI/KMS, certificate rotation, secrets management, network policy, observability, hardened images, vulnerability management, access control, incident response, backup/recovery, and deployment-specific threat analysis.
