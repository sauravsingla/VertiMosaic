# VertiMosaic

**CPU-first vertical federated learning for heterogeneous tabular data.**

VertiMosaic is an open-source research framework for vertical federated learning (VFL) on aligned tabular entities. It is designed around protocol correctness, reproducibility, provenance, auditable communication, explicit privacy boundaries, and CPU-friendly reference implementations.

[Get started](quickstart.md){ .md-button .md-button--primary }
[Architecture](architecture.md){ .md-button }
[GitHub](https://github.com/sauravsingla/VertiMosaic){ .md-button }

!!! note "Privacy posture"
    Raw-feature locality is a design property of VertiMosaic. It is **not** presented as equivalent to end-to-end cryptographic privacy. Privacy guarantees are scoped to the exact mechanism, protected release, observer, and threat model documented for each experiment.

## Why VertiMosaic

<div class="grid cards" markdown>

-   :material-vector-arrange-below:{ .lg .middle } **Strict entity alignment**

    ---

    Order-sensitive entity identifiers and digests can reject equal-length but misaligned party partitions before training, validation, or inference.

-   :material-cpu-64-bit:{ .lg .middle } **CPU-first research**

    ---

    Reference logistic-regression and histogram-GBDT VFL implementations are designed to reproduce on developer-class CPU environments without requiring a GPU.

-   :material-shield-lock-outline:{ .lg .middle } **Explicit privacy boundaries**

    ---

    Optional clipped-Gaussian releases, secure aggregation, additive sharing, PSI, and Paillier primitives are documented without silently upgrading the default protocol's privacy claims.

-   :material-test-tube:{ .lg .middle } **Release-grade evidence**

    ---

    Reproducibility bundles, hashes, environment metadata, privacy audits, communication measurements, release assets, and clean-room reproduction tooling make results inspectable.

</div>

## Current research surface

| Area | Current scope |
|---|---|
| Federation | Vertical federated learning |
| Reference models | `VFLLogisticRegression`, `VFLHistGBDT` |
| Compute | CPU supported; GPU not required |
| Exact multi-source public linkage | NPI-linked Open Payments + NPPES + CMS Care Compare/provider data |
| Additional linked benchmarks | MovieLens 1M, UCI Credit, authorized local IEEE-CIS |
| Cross-industry benchmark | Public-source, explicitly semi-synthetic linkage |
| Protected research path | PSI alignment + clipped-Gaussian residual releases with scoped accounting |
| External comparison | Deterministic exchange/normalization contract for third-party VFL frameworks |
| Reproduction | PyPI install path, evidence bundles, attestation schema, validation tooling |

## Documentation map

- **[Quick Start](quickstart.md)** — install the package and run the first deterministic CPU experiment.
- **[Architecture](architecture.md)** — understand parties, models, transports, evaluation, and reporting boundaries.
- **[Protocol](protocol.md)** — entity alignment, message flow, missing-party behavior, and protocol invariants.
- **[Privacy](privacy_boundaries.md)** — what the framework protects, what it does not, and how optional backends change the boundary.
- **[Benchmarks](benchmarks.md)** — distinguish exact linkage, vertical partitioning, and semi-synthetic cross-domain experiments.
- **[Reproduction](reproducibility.md)** — produce inspectable evidence bundles and independently validate a release.
- **[API Reference](api.md)** — primary public classes and research entry points.

## Version

The documentation targets the current **v0.3.x** research surface. For immutable artifacts and measured release evidence, use the corresponding GitHub release rather than assuming the `main` branch and a published tag are identical.
