<h1 align="center">VertiMosaic</h1>

<p align="center">
  <b>Train one model across organizations without pooling their raw tabular features — on CPU.</b>
</p>

<p align="center">
  Vertical federated learning for aligned entities, heterogeneous feature sets, reproducible experiments, and explicit privacy boundaries.
</p>

<p align="center">
  <a href="https://github.com/sauravsingla/VertiMosaic/actions/workflows/ci.yml"><img src="https://github.com/sauravsingla/VertiMosaic/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://github.com/sauravsingla/VertiMosaic/actions/workflows/security.yml"><img src="https://github.com/sauravsingla/VertiMosaic/actions/workflows/security.yml/badge.svg?branch=main" alt="Security"></a>
  <a href="https://github.com/sauravsingla/VertiMosaic/actions/workflows/package.yml"><img src="https://github.com/sauravsingla/VertiMosaic/actions/workflows/package.yml/badge.svg?branch=main" alt="Package"></a>
  <a href="https://pypi.org/project/vertimosaic/"><img src="https://img.shields.io/pypi/v/vertimosaic" alt="PyPI version"></a>
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue" alt="Python 3.11 | 3.12">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue" alt="Apache-2.0"></a>
  <a href="https://huggingface.co/datasets/sauravsingla08/VertiMosaic-VFL-Benchmark"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E" alt="Hugging Face Dataset"></a>
  <a href="https://huggingface.co/spaces/sauravsingla08/VertiMosaic"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Space-FFD21E" alt="Hugging Face Space"></a>
</p>

## What VertiMosaic does

Many organizations can describe the **same entities** while holding **different feature columns**. A bank may know payment behavior, a telecom provider service behavior, an insurer risk signals, and a retailer purchase behavior.

VertiMosaic lets researchers and ML engineers **build, benchmark, and audit vertical federated learning workflows while each passive party keeps its raw feature table local**.

### 30-second mental model

```text
Same aligned entities

Bank                    Telecom                 Insurance               Retail
[age, income, target]   [usage, tenure]         [claims, risk]          [spend, frequency]
        │                     │                       │                        │
        └──────────── party-local computation + protocol messages ────────────┘
                                      │
                                joint VFL model
```

The reference protocols exchange model- and training-related signals instead of pooling passive-party raw feature matrices at the active party. **Raw-feature locality is not the same as end-to-end cryptographic privacy**, so privacy guarantees and non-guarantees are documented explicitly.

## Try it in under a minute

VertiMosaic supports Python **3.11 and 3.12** and does not require a GPU.

```bash
python -m pip install vertimosaic
vertimosaic demo --rows 2000 --seed 42
```

The demo runs a CPU-friendly smoke experiment and writes a reproducibility bundle under `runs/<run_id>/`.

To run directly from source:

```bash
git clone https://github.com/sauravsingla/VertiMosaic.git
cd VertiMosaic
python -m pip install -e .
vertimosaic demo --rows 2000 --seed 42
```

## Inspect measured evidence

VertiMosaic avoids hand-written headline benchmark numbers. The **v0.3.0 release** publishes machine-generated evidence from the release source so results can be inspected independently.

| Evidence | What it contains |
|---|---|
| [`canonical-results.json`](https://github.com/sauravsingla/VertiMosaic/releases/download/v0.3.0/canonical-results.json) | Canonical release-result summary |
| [`formal_comparison.md`](https://github.com/sauravsingla/VertiMosaic/releases/download/v0.3.0/formal_comparison.md) | Centralized, single-party, and VFL comparison output |
| [`formal_comparison.csv`](https://github.com/sauravsingla/VertiMosaic/releases/download/v0.3.0/formal_comparison.csv) | Machine-readable comparison table |
| [`remote_transport.json`](https://github.com/sauravsingla/VertiMosaic/releases/download/v0.3.0/remote_transport.json) | Serialized remote-transport measurements |
| [`privacy_audit.json`](https://github.com/sauravsingla/VertiMosaic/releases/download/v0.3.0/privacy_audit.json) | Reproducible privacy-attack baseline |
| [`advanced_privacy_audit.json`](https://github.com/sauravsingla/VertiMosaic/releases/download/v0.3.0/advanced_privacy_audit.json) | Advanced privacy-audit output |
| [Full v0.3.0 release](https://github.com/sauravsingla/VertiMosaic/releases/tag/v0.3.0) | Manifest, hashes, environment capture, preprint, and packaged evidence |

Maintainer-generated release evidence demonstrates reproducibility of the maintained pipeline; it is **not the same as independent third-party reproduction**. Independent reproduction is supported separately under [`reproductions/`](reproductions/) and [`docs/independent_reproduction.md`](docs/independent_reproduction.md).

## Why VertiMosaic?

- **CPU-first** — runs without dedicated accelerators and fits laptops, CI, and GitHub-hosted research workflows.
- **Built for tabular VFL** — includes reference logistic and histogram-GBDT protocols rather than only a generic distributed-training wrapper.
- **Strict entity alignment** — research paths can reject equal-length but differently ordered party rows instead of silently treating them as aligned.
- **Auditable communication** — protocol messages, payload accounting, and a separate serialized HTTP benchmark path are measurable.
- **Linked public benchmarks** — includes exact-NPI multi-source provider linkage and exact multi-table MovieLens linkage.
- **Explicit privacy boundaries** — distinguishes raw-feature locality from stronger DP, PSI, MPC, HE, secure-aggregation, or malicious-party guarantees.
- **Reproducibility first** — deterministic runs, release evidence, environment capture, hashes, benchmark outputs, and independent-reproduction schemas.

## Pick your path

| If you want to... | Start here |
|---|---|
| Understand VFL quickly | [`tutorials/01_first_vfl_10_minutes.md`](tutorials/01_first_vfl_10_minutes.md) |
| Run the basic demo | `vertimosaic demo --rows 2000 --seed 42` |
| Compare centralized / single-party / VFL paths | `vertimosaic-benchmark-matrix` |
| Run a public exact-linked benchmark | `vertimosaic-linked-movielens` |
| Inspect privacy limitations | [`docs/privacy_boundaries.md`](docs/privacy_boundaries.md) |
| Reproduce the maintained evidence bundle | `vertimosaic-reproduce` |
| Try separate services with HTTPS/mTLS | [`deploy/compose/README.md`](deploy/compose/README.md) |
| Explore the live project surface | [Hugging Face Space](https://huggingface.co/spaces/sauravsingla08/VertiMosaic) |

## At a glance

| Property | Current scope |
|---|---|
| Federation | Vertical federated learning |
| Reference models | `VFLLogisticRegression`, `VFLHistGBDT` |
| Compute | CPU supported; GPU not required |
| Raw passive feature tables pooled during VFL | No |
| Entity alignment | Order-sensitive validation in strict research paths |
| Public linked benchmarks | NPI multi-source, MovieLens multi-table, UCI exact-row; optional local IEEE-CIS |
| Privacy research | PSI, clipped Gaussian/zCDP, secure-sum, additive sharing, Paillier adapters |
| Reproducibility | Run bundles, release evidence, hashes, environment capture, reproduction schemas |

## Architecture

<p align="center">
  <img src="assets/vertimosaic-architecture.svg" alt="VertiMosaic architecture" width="100%">
</p>

The active party owns the target. Passive parties retain their raw feature matrices and return protocol-specific local outputs. `VFLLogisticRegression` exchanges local logits/residual-related signals for party-local gradient updates. `VFLHistGBDT` exchanges target-derived gradient/Hessian signals and aggregate histogram candidate statistics while numeric split thresholds and routing state remain with the owning party.

`InMemoryTransport` provides deterministic protocol simulation and message auditing. `RemoteHTTPTransport` provides a separate serialized HTTP path with bounded requests/responses, authorization, replay controls, idempotency limits, rate limiting, optional compressed NumPy transport, and signed-message support. The Docker Compose example adds separate services, party-local volumes, HTTPS/mTLS, bearer authorization, health checks, and deterministic sample data.

See [`docs/architecture.md`](docs/architecture.md) and [`docs/protocol.md`](docs/protocol.md).

## Reference protocols

### `VFLLogisticRegression`

A first-principles NumPy reference protocol. Each party computes logits and gradients using only its local feature matrix. Passive logits and residual-related signals pass through the explicit message/transport abstraction.

The model records the parties present during training. At inference, omission of a trained party is an error by default; the research-only `zero_contribution` fallback must be selected explicitly.

### `VFLHistGBDT`

A vertical histogram-gradient-boosting reference implementation. Each party fits and retains its training-derived quantile bins. Passive parties compute aggregate histogram candidates from received gradient/Hessian signals and return aggregate statistics plus opaque feature/bin references. The owning party resolves private numeric thresholds and performs routing locally.

Centralized models are included only as **non-federated baselines** and must not be described as VFL.

## Entity alignment

Strict research paths bind ordered pseudonymous entity identifiers to party partitions and derive order-sensitive digests. Training, validation, and inference can reject party collections whose entity sets or ordering do not match exactly.

This is stronger than checking row counts alone: two parties can have the same number of rows while referring to different entities or a different order.

See [`src/vertimosaic/alignment/`](src/vertimosaic/alignment/) and [`docs/protocol.md`](docs/protocol.md).

## Protected logistic research path

VertiMosaic v0.3.0 includes an explicitly scoped protected logistic workflow in [`src/vertimosaic/privacy/protected_logistic.py`](src/vertimosaic/privacy/protected_logistic.py). It can:

1. intersect party entity sets through a configured PSI backend;
2. canonicalize parties to one exact ordered entity sequence;
3. bind strict entity metadata to aligned partitions; and
4. train `VFLLogisticRegression` with `ClippedGaussianDPBackend` applied to active-to-passive residual releases.

The resulting privacy report is deliberately narrow. PSI covers set intersection according to the selected PSI backend's threat model, while clipped-Gaussian accounting covers the protected residual-release family only. **Logits, model parameters, timing, other protocol messages, routing information, and transport metadata remain outside that composite guarantee.**

See [`docs/protected_logistic.md`](docs/protected_logistic.md), [`docs/privacy_boundaries.md`](docs/privacy_boundaries.md), and [`docs/threat_model.md`](docs/threat_model.md).

## Benchmarks and linkage categories

VertiMosaic keeps linkage categories explicit so reproducibility is not confused with stronger real-world linkage evidence.

| Benchmark | Linkage category | Purpose |
|---|---|---|
| NPI provider benchmark | Real exact multi-source public linkage | Links Open Payments, NPPES, and CMS provider data using NPI |
| MovieLens 1M | Public exact multi-table linkage | Same service users linked across users, ratings, and movies tables |
| IEEE-CIS | Authorized local exact linkage | Optional local transaction/identity linkage on `TransactionID` |
| UCI Credit | Exact-row vertical partition | Public sanity benchmark with disjoint feature columns |
| Bank / Telecom / Insurance / Retail | Explicitly semi-synthetic cross-domain linkage | Controlled cross-industry research setting |
| Synthetic scale / overlap | Controlled synthetic study | Scale, overlap, dropout, drift, noise, and distributed-signal experiments |

The four-industry benchmark does **not** claim that its public source datasets describe the same real individuals. Full definitions, sources, and limitations are documented in [`docs/linked_benchmarks.md`](docs/linked_benchmarks.md), [`docs/npi_linked_benchmark.md`](docs/npi_linked_benchmark.md), and [`docs/linkage.md`](docs/linkage.md).

## Evaluation and reproducibility

Entity-level splits default to **70% train / 15% validation / 15% test**. Threshold selection uses validation predictions only.

Evaluation includes ROC-AUC, PR-AUC, precision, recall, F1, balanced accuracy, log loss, Brier score, calibration error, confusion counts, deterministic bootstrap intervals, and paired bootstrap differences.

`vertimosaic-benchmark-matrix` compares:

- centralized all-feature **non-federated** logistic regression;
- Bank-only **non-federated** logistic regression;
- `VFLLogisticRegression`; and
- `VFLHistGBDT`.

The formal matrix records utility, training/inference wall-clock time, sampled process RSS, protocol payload bytes/message counts, robustness fields, and a reproducible privacy-attack baseline. Protocol payload accounting is not a packet capture.

Every standard run bundle records configuration, dataset/linkage provenance, environment and dependency versions, predictions, metrics, training history, communication metadata, hashes, timestamps, seed, and Git state. `vertimosaic-reproduce` creates a clean reproduction evidence bundle with environment metadata and a stable result digest.

## External VFL comparators

VertiMosaic provides a deterministic export/import contract for comparison with established VFL frameworks such as **FATE** and **SecretFlow**:

- [`scripts/export_comparator_benchmark.py`](scripts/export_comparator_benchmark.py) freezes aligned arrays, targets, entity IDs, splits, and SHA-256 metadata;
- [`scripts/run_external_comparator.py`](scripts/run_external_comparator.py) normalizes measured external-framework output into the repository comparison schema;
- [`docs/external_comparators.md`](docs/external_comparators.md) documents the contract and claim boundaries.

A wrapper or exchange format is **not** treated as a measured third-party result. FATE/SecretFlow rows remain pending until those frameworks are actually run against the frozen exchange bundle.

## Privacy and security boundary

The default VFL protocols provide raw-feature locality, party-local preprocessing/computation, explicit message boundaries, and auditable research transports. They do **not** automatically provide end-to-end differential privacy, PSI, MPC, homomorphic encryption, secure aggregation, collusion resistance, or malicious-party security.

Sensitive derived signals—including gradients, Hessians, logits, residuals, entity membership, routing information, and split statistics—may leak information depending on the observer and protocol.

Optional, explicitly scoped research mechanisms include:

- `ClippedGaussianDPBackend` for message-level clipping, Gaussian noise, zCDP composition, and `(epsilon, delta)` conversion;
- `PairwiseMaskSecureAggregation` as an honest-but-curious secure-sum reference primitive;
- `AdditiveSecretSharingSum` for additive-share sum experiments;
- `OpenMinedPSIBackend` as an optional PSI adapter; and
- `PaillierHomomorphicSum` for bounded additive homomorphic-sum experiments.

None of these primitives should be interpreted as a blanket **"secure VFL"** guarantee. See [`docs/privacy_backends.md`](docs/privacy_backends.md), [`docs/privacy_experiments.md`](docs/privacy_experiments.md), [`docs/privacy_boundaries.md`](docs/privacy_boundaries.md), and [`docs/threat_model.md`](docs/threat_model.md).

## What is new in v0.3.0

- order-sensitive entity alignment;
- explicit missing-party behavior in `VFLLogisticRegression`;
- protected logistic research path with PSI-based alignment and scoped clipped-Gaussian residual releases;
- exact-NPI multi-source public benchmark builder;
- external-comparator exchange contract for measured FATE/SecretFlow comparisons;
- independent-reproduction evidence schemas and validation tooling;
- four focused tutorials and a separate-service Docker Compose HTTPS/mTLS example; and
- immutable release-evidence assets with results, privacy audits, environment metadata, hashes, and reproduction artifacts.

Latest release: **[VertiMosaic v0.3.0](https://github.com/sauravsingla/VertiMosaic/releases/tag/v0.3.0)**.

## Installation and tutorials

Development and verification tooling:

```bash
python -m pip install -e ".[dev]"
```

Optional PSI and Paillier adapters:

```bash
python -m pip install -e ".[privacy-crypto]"
```

Tutorials:

- [`tutorials/01_first_vfl_10_minutes.md`](tutorials/01_first_vfl_10_minutes.md) — first VFL run in about 10 minutes
- [`tutorials/02_two_party_logistic.md`](tutorials/02_two_party_logistic.md) — strict two-party logistic VFL
- [`tutorials/03_histogram_gbdt.md`](tutorials/03_histogram_gbdt.md) — vertical histogram GBDT
- [`tutorials/04_privacy_leakage_and_mitigation.md`](tutorials/04_privacy_leakage_and_mitigation.md) — leakage, audits, and scoped mitigations

Useful research commands:

| Goal | Command |
|---|---|
| Formal centralized / single-party / VFL comparison | `vertimosaic-benchmark-matrix` |
| Public exact-linked MovieLens benchmark | `vertimosaic-linked-movielens` |
| Public UCI exact-row vertical partition | `vertimosaic-linked-uci` |
| Separate-process serialized transport benchmark | `vertimosaic-remote-benchmark` |
| Advanced privacy research audit | `vertimosaic-advanced-privacy-audit` |
| Clean reproduction evidence bundle | `vertimosaic-reproduce` |

## Verification and portability

Core checks:

```bash
ruff check .
ruff format --check .
mypy src/vertimosaic
pytest -q --cov=vertimosaic --cov-report=term-missing
bandit -r src -q
pip-audit
python -m build
twine check dist/*
```

CI exercises Python 3.11/3.12, protocol-critical coverage, package validation, privacy audits, remote transport measurement, CodeQL, dependency audit/SBOM generation, optional crypto backends, clean-room reproduction, external-data integration, and portability smoke checks on Linux ARM64, macOS, and Windows.

## Research report and documentation

The working technical report is [`paper/preprint.md`](paper/preprint.md), with the experiment plan in [`paper/experiment_manifest.md`](paper/experiment_manifest.md).

Key documentation:

- [`docs/architecture.md`](docs/architecture.md) — architecture and component boundaries
- [`docs/protocol.md`](docs/protocol.md) — protocol details and alignment behavior
- [`docs/formal_benchmark.md`](docs/formal_benchmark.md) — benchmark matrix and interpretation boundaries
- [`docs/linked_benchmarks.md`](docs/linked_benchmarks.md) — linked-benchmark taxonomy
- [`docs/npi_linked_benchmark.md`](docs/npi_linked_benchmark.md) — exact-NPI benchmark
- [`docs/external_comparators.md`](docs/external_comparators.md) — comparator exchange and measurement contract
- [`docs/protected_logistic.md`](docs/protected_logistic.md) — protected logistic research path
- [`docs/privacy_boundaries.md`](docs/privacy_boundaries.md) — privacy claim boundaries
- [`docs/threat_model.md`](docs/threat_model.md) — adversary and non-goals
- [`docs/reproducibility.md`](docs/reproducibility.md) — experiment reproducibility
- [`docs/independent_reproduction.md`](docs/independent_reproduction.md) — external reproduction procedure
- [`docs/release_evidence.md`](docs/release_evidence.md) — immutable release evidence
- [`docs/limitations.md`](docs/limitations.md) — known limitations and non-claims

## Reproduce or contribute

The most useful external signal for this project is an independent run.

1. Install the release and run the demo or a benchmark.
2. Keep the generated run bundle and environment details.
3. Submit a [reproduction report](https://github.com/sauravsingla/VertiMosaic/issues/new?template=reproduction_report.md) or open an issue with a minimal reproducible case.

General contribution guidance is in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Citation

If you use VertiMosaic in research or development, cite the software using [`CITATION.cff`](CITATION.cff). GitHub also exposes a **Cite this repository** action from that file.

```text
VertiMosaic
Saurav Singla
Version 0.3.0
Apache-2.0
https://github.com/sauravsingla/VertiMosaic
```

## License

VertiMosaic source code is licensed under the **Apache License 2.0**. Dataset licenses and provider terms remain separate; see [`DATA_LICENSES.md`](DATA_LICENSES.md).