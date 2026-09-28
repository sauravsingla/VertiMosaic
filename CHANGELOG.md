# Changelog

## 0.3.0 — 2026-09-28

- Added order-sensitive entity IDs/digests and strict alignment validation across fit,
  validation and inference paths so row permutations cannot silently train as aligned data.
- Made VFL logistic missing-party inference explicit with `error` as the default policy and
  opt-in zero-contribution behavior for controlled robustness research.
- Integrated clipped-Gaussian residual releases and privacy accounting into an actual logistic
  VFL path, plus a scoped PSI alignment + protected logistic research configuration.
- Added a deterministic external-framework benchmark exchange and normalized comparator contract
  for measured FATE, SecretFlow or other VFL implementations without fabricating external results.
- Added independent-reproduction submission, attestation and validation tooling so unaffiliated
  researchers can publish machine-verifiable reproduction evidence.
- Added an exact-NPI public multi-source benchmark builder spanning Open Payments, NPPES and CMS
  provider data with conservative linkage, licensing and redistribution boundaries.
- Expanded limitations and non-guarantees covering entity linkage, honest-but-curious assumptions,
  collusion, gradient/Hessian and routing leakage, poisoning, scalability, fairness/domain shift,
  deployment boundaries and compliance non-claims.
- Added canonical measured-result indexing and release-evidence packaging with environment,
  checksums and generated publication artifacts.
- Added a technical preprint plus new research questions for external comparators, protected VFL
  and exact-NPI linkage.
- Added four focused first-user tutorials and a deterministic multi-service Docker Compose example
  with separate party services, HTTPS/mTLS options, bearer authentication and health checks.
- Fixed Hugging Face dataset publication validation to use single-threaded PyArrow parquet reads,
  avoiding a Linux CI interpreter-shutdown abort after otherwise successful validation.

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
- Added Windows, macOS, and Linux ARM64 portability smoke CI, optional cryptographic-backend
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
