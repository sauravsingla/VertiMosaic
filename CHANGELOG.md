# Changelog

## 0.2.0 — 2026-09-28

- Added a formal four-protocol benchmark matrix covering centralized, single-party, VFL
  logistic, and VFL histogram GBDT comparisons with utility, runtime, sampled peak RSS,
  communication and empirical privacy-leakage measurements.
- Added an exact-row, disjoint-column UCI Credit public linked sanity benchmark with
  training-only preprocessing, source checksum, pseudonymous predictions and claim boundaries.
- Added a genuinely exact public multi-table MovieLens 1M benchmark using observed `UserID`
  linkage across user and rating records plus `MovieID` linkage to movie metadata. Source data
  are downloaded from GroupLens at runtime and are not redistributed by VertiMosaic.
- Hardened the reference remote transport with bounded TTL/LRU idempotency, request-digest
  binding, sender/message authorization, rotating bearer-token support, replay-window timestamps,
  local or external rate limiting, HMAC-signed request/response bodies, bounded responses and
  optional compressed NumPy serialization.
- Added a separate-process remote transport benchmark reporting measured serialized HTTP
  application bytes and latency while explicitly excluding unobservable TLS/TCP/IP framing.
- Added a sensitivity-enforcing clipped Gaussian release backend with zCDP accounting while
  retaining explicit message-level rather than end-to-end DP claim boundaries.
- Added narrow cryptographic research backends for pairwise-mask secure aggregation and additive
  secret-sharing sums, plus optional OpenMined PSI and Paillier homomorphic-sum adapters.
- Expanded empirical privacy evaluation with gradient/label leakage, routing exposure and
  message-to-feature reconstruction baselines covering both passive- and active-party observers.
- Added wheel-only clean-room reproduction tooling and an independent reproduction evidence
  bundle command with exact environment/version capture and SHA-256 manifests.
- Added Windows, macOS ARM64 and Linux ARM64 portability smoke CI, optional cryptographic-backend
  CI, CodeQL, OpenSSF Scorecard, Dependabot and Sigstore-backed GitHub artifact attestations.
- Added a guarded release branch/tag workflow, CODEOWNERS and documented main-branch integrity
  settings; live branch protection remains a repository-administration setting and must be
  verified independently of source files.
- Added a v0.1.x freeze/release archival policy, Zenodo metadata and external-contribution
  issue scaffolding.

## 0.1.0 — 2026-09-27

- Established VertiMosaic project identity and CPU-only package.
- Added reference VFL logistic regression and vertical histogram GBDT research protocols.
- Added metadata-only communication audit and privacy-boundary tests.
- Added synthetic and externally grounded benchmark infrastructure, linkage, provenance, reproducibility, robustness studies, uncertainty estimation, reporting, and CI/security scaffolding.
- Hardened dataset provenance so every source record exposes the full machine-readable schema and unknown provider values remain explicit rather than fabricated.
- Added runtime OpenML license verification through the official metadata API and wired it into the scheduled/manual external-data integration workflow.
