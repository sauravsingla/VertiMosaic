# Changelog

## 0.2.0 — development

- Added a formal four-protocol benchmark matrix covering centralized, single-party, VFL
  logistic, and VFL histogram GBDT comparisons with utility, runtime, memory,
  communication and empirical privacy-leakage measurements.
- Added an exact-row, disjoint-column UCI Credit public linked sanity benchmark with
  training-only preprocessing, source checksum, pseudonymous predictions and claim boundaries.
- Hardened the reference remote transport with bounded TTL/LRU idempotency, sender/message
  authorization, replay-window timestamps, per-sender rate limiting, bounded responses and
  optional compressed NumPy serialization.
- Added an optional Gaussian release backend with zCDP accounting. This is mechanism-level
  research support and is not an automatic end-to-end DP guarantee.
- Added wheel-only clean-room reproduction tooling and an independent reproduction evidence
  bundle command with exact environment/version capture and SHA-256 manifests.
- Added a v0.1.x freeze/release archival policy, Zenodo metadata and external-contribution
  issue scaffolding.

## 0.1.0 — development

- Established VertiMosaic project identity and CPU-only package.
- Added reference VFL logistic regression and vertical histogram GBDT research protocols.
- Added metadata-only communication audit and privacy-boundary tests.
- Added synthetic and externally grounded benchmark infrastructure, linkage, provenance, reproducibility, robustness studies, uncertainty estimation, reporting, and CI/security scaffolding.
- Hardened dataset provenance so every source record exposes the full machine-readable schema and unknown provider values remain explicit rather than fabricated.
- Added runtime OpenML license verification through the official metadata API and wired it into the scheduled/manual external-data integration workflow.
