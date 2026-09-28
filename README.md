<h1 align="center">VertiMosaic</h1>

<p align="center">
  <b>CPU-first Vertical Federated Learning for Heterogeneous Tabular Data</b>
</p>

<p align="center">
  A reproducible research framework for organizations that hold complementary feature columns about the same or overlapping entities while keeping raw party feature tables local.
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

VertiMosaic is an open-source **vertical federated learning (VFL)** research framework for tabular data. Different parties retain different feature columns for aligned entities, and the reference VFL protocols request party-local computations instead of pooling passive-party raw feature matrices at the active party.

The project focuses on **reproducibility, provenance, communication measurement, explicit privacy boundaries, and CPU-friendly reference implementations**. Raw-feature locality is a design property; it is not presented as equivalent to end-to-end cryptographic privacy.

## At a glance

| Property | Current scope |
|---|---|
| Federation | Vertical federated learning |
| Reference models | `VFLLogisticRegression`, `VFLHistGBDT` |
| Active party in the cross-industry benchmark | Bank, with target |
| Passive parties | Telecom, Insurance, Retail |
| Raw passive feature tables pooled during VFL | No |
| Compute | CPU supported; GPU not required |
| Public exact-linked benchmark | MovieLens 1M multi-table linkage |
| Additional linked benchmarks | UCI exact-row vertical partition; authorized local IEEE-CIS transaction/identity linkage |
| Cross-industry benchmark | Public-source, explicitly semi-synthetic linkage |
| Default privacy posture | Research protocol with raw-feature locality; not end-to-end cryptographic VFL |
| Optional privacy research mechanisms | Clipped Gaussian/zCDP releases, secure-sum and additive-sharing primitives, PSI and Paillier adapters |

## Architecture

<p align="center">
  <img src="assets/vertimosaic-architecture.svg" alt="VertiMosaic architecture" width="100%">
</p>

The active party owns the target. Passive parties retain their raw feature matrices and return only protocol-specific local outputs. `VFLLogisticRegression` exchanges local logits/residual-related signals for party-local gradient updates. `VFLHistGBDT` exchanges target-derived gradient/Hessian signals and aggregate histogram candidate statistics while numeric split thresholds and routing state remain with the owning party.

`InMemoryTransport` provides deterministic protocol simulation and message auditing. `RemoteHTTPTransport` provides a separate serialized HTTP path with bounded requests/responses, authorization, replay controls, idempotency limits, rate limiting, optional compressed NumPy transport, and signed-message support. These transport controls do not turn the default learning protocol into malicious-secure VFL.

## Installation

VertiMosaic supports Python **3.11 and 3.12**.

Install the latest published release:

```bash
python -m pip install vertimosaic
```

For the current repository state:

```bash
git clone https://github.com/sauravsingla/VertiMosaic.git
cd VertiMosaic
python -m pip install --upgrade pip
python -m pip install -e .
```

For development and verification tooling:

```bash
python -m pip install -e ".[dev]"
```

Optional PSI and Paillier adapters:

```bash
python -m pip install -e ".[privacy-crypto]"
```

## Quick start

```bash
vertimosaic --help
vertimosaic demo --rows 2000 --seed 42
```

The demo is a smoke experiment and uses **100 bootstrap replicates**. Normal research experiment and benchmark paths use **1,000 bootstrap replicates** where practical. Standard experiment outputs are written as reproducibility bundles under `runs/<run_id>/`.

### Useful research commands

| Goal | Command |
|---|---|
| Formal centralized / single-party / VFL comparison | `vertimosaic-benchmark-matrix` |
| Public exact-linked MovieLens benchmark | `vertimosaic-linked-movielens` |
| Public UCI exact-row vertical partition | `vertimosaic-linked-uci` |
| Separate-process serialized transport benchmark | `vertimosaic-remote-benchmark` |
| Advanced privacy research audit | `vertimosaic-advanced-privacy-audit` |
| Clean reproduction evidence bundle | `vertimosaic-reproduce` |
| Cross-industry / synthetic CLI workflows | `vertimosaic --help` |

## Reference protocols

### `VFLLogisticRegression`

A first-principles NumPy reference protocol. Each party computes logits and gradients using only its local feature matrix. Passive logits and residual-related signals pass through the explicit message/transport abstraction.

### `VFLHistGBDT`

A vertical histogram-gradient-boosting reference implementation. Each party fits and retains its own training-derived quantile bins. Passive parties compute aggregate histogram candidates from received gradient/Hessian signals and return aggregate statistics plus opaque feature/bin references. The active party selects a split; the owning party resolves the private numeric threshold and performs routing locally. Validation and test rows reuse training-derived routing state rather than refitting held-out bins.

Centralized models are provided only as **non-federated baselines**. They deliberately pool selected feature columns and must not be described as VFL.

## Benchmarks and linkage categories

VertiMosaic keeps linkage categories explicit so reproducibility is not confused with stronger real-world linkage evidence.

### 1. MovieLens 1M — public exact multi-table linkage

`vertimosaic-linked-movielens` downloads MovieLens 1M directly from GroupLens and links its observed `users.dat`, `ratings.dat`, and `movies.dat` identifiers. The active side uses user demographics; the passive side derives historical rating/genre behavior from the earlier portion of each user's timeline; the target is derived from later rating events that are excluded from passive features.

This is **genuine exact linkage across public tables describing the same service users**. It is still a single-service dataset, not cross-organization entity resolution and not evidence of PSI.

### 2. IEEE-CIS — authorized local exact linkage

The optional IEEE-CIS path joins locally authorized transaction and identity files on exact `TransactionID` intersection. VertiMosaic does not download or redistribute those competition files.

```bash
vertimosaic prepare-ieee-cis \
  --transaction /path/to/train_transaction.csv \
  --identity /path/to/train_identity.csv

vertimosaic run-ieee-cis \
  --transaction /path/to/train_transaction.csv \
  --identity /path/to/train_identity.csv \
  --model logistic \
  --seed 42
```

### 3. UCI Credit — exact-row vertical partition

`vertimosaic-linked-uci` retrieves UCI Default of Credit Card Clients at runtime and partitions disjoint source columns between active and passive parties while preserving exact entity rows. It is a useful public linkage sanity benchmark, but the two partitions originate from one source dataset and therefore do not demonstrate cross-organization entity resolution.

### 4. Four-industry benchmark — semi-synthetic cross-domain linkage

The Bank/Telecom/Insurance/Retail benchmark uses real public source-domain data but the four sources do **not** describe the same real individuals. They are independently preprocessed and connected by an explicitly documented research linkage mechanism.

| Party | Public source | Role |
|---|---|---|
| Bank | UCI Default of Credit Card Clients (350) | Financial/payment behavior + observed target |
| Telecom | UCI Iranian Churn (563) | Telecom/service behavior |
| Insurance | OpenML `freMTPL2freq` + `freMTPL2sev` (41214/41215) | Claims/risk behavior |
| Retail | UCI Online Retail (352) | Purchase behavior aggregated to customer level |

`observed_target_external` uses the Bank target with target-blind external linkage. `distributed_signal_external` links transformed source-domain features first and generates the disclosed semi-synthetic target only after the linked profiles are formed.

### 5. Controlled synthetic and overlap studies

`synthetic_scale` supports controlled scaling, overlap, dropout, drift, noise, and distributed-signal experiments. `vertimosaic overlap` evaluates **100%, 90%, 75%, 50%, and 25%** Bank-to-passive overlap while preserving deterministic entity splits and reporting coverage, utility, and communication separately for each strategy.

For full linkage definitions and limitations, see [`docs/linked_benchmarks.md`](docs/linked_benchmarks.md) and [`docs/linkage.md`](docs/linkage.md).

## Evaluation, communication, and reproducibility

Entity-level splits default to **70% train / 15% validation / 15% test**. Threshold selection uses validation predictions only.

Evaluation includes ROC-AUC, PR-AUC, precision, recall, F1, balanced accuracy, log loss, Brier score, calibration error, confusion counts, deterministic bootstrap intervals, and paired bootstrap differences.

`vertimosaic-benchmark-matrix` compares:

- centralized all-feature **non-federated** logistic regression;
- Bank-only **non-federated** logistic regression;
- `VFLLogisticRegression`; and
- `VFLHistGBDT`.

The formal matrix records utility, training/inference wall-clock time, sampled process RSS, protocol payload bytes/message counts, robustness fields, and a reproducible privacy-attack baseline. Protocol payload accounting is not a packet capture.

`vertimosaic-remote-benchmark` separately exercises the real serialized HTTP path through a spawned relay process and records logical payload bytes plus transport-level request/response measurements and elapsed time. TLS record bytes are not claimed because the underlying HTTP client does not expose them.

Every standard run bundle records configuration, dataset/linkage provenance, environment and dependency versions, predictions, metrics, training history, communication metadata, hashes, timestamps, seed, and Git state. `vertimosaic-reproduce` creates an additional clean reproduction evidence bundle with environment metadata and a stable result digest.

## Privacy and security boundary

The default VFL protocols provide raw-feature locality, party-local preprocessing/computation, explicit message boundaries, and auditable research transports. They do **not** automatically provide end-to-end differential privacy, PSI, MPC, homomorphic encryption, secure aggregation, collusion resistance, or malicious-party security.

Sensitive derived signals—including gradients, Hessians, logits, residuals, entity membership, routing information, and split statistics—may leak information depending on the observer and protocol.

VertiMosaic v0.2.0 also includes **optional, explicitly scoped research mechanisms**:

- `ClippedGaussianDPBackend` for message-level L2 clipping, Gaussian noise, zCDP composition, and `(epsilon, delta)` conversion;
- `PairwiseMaskSecureAggregation` as an honest-but-curious secure-sum reference primitive;
- `AdditiveSecretSharingSum` for additive-share sum experiments;
- `OpenMinedPSIBackend` as an optional PSI adapter; and
- `PaillierHomomorphicSum` for bounded additive homomorphic-sum experiments.

None of these primitives is silently enabled in the default VFL algorithms, and their presence must not be interpreted as a blanket "secure VFL" guarantee. See [`docs/threat_model.md`](docs/threat_model.md), [`docs/privacy_boundaries.md`](docs/privacy_boundaries.md), [`docs/privacy_backends.md`](docs/privacy_backends.md), and [`docs/privacy_experiments.md`](docs/privacy_experiments.md).

## Verification and portability

Core verification:

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

CI exercises Python 3.11/3.12, protocol-critical coverage, paper/reproduction smoke paths, privacy audits, remote transport measurement, package validation, CodeQL, dependency audit/SBOM generation, optional crypto backends, and portability smoke checks on Linux ARM64, macOS, and Windows. Protocol-critical modules target at least **90% coverage**.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — architecture and component boundaries
- [`docs/protocol.md`](docs/protocol.md) — reference protocol details
- [`docs/formal_benchmark.md`](docs/formal_benchmark.md) — benchmark matrix and interpretation boundaries
- [`docs/linked_benchmarks.md`](docs/linked_benchmarks.md) — exact-linked and semi-synthetic benchmark taxonomy
- [`docs/datasets.md`](docs/datasets.md) — source datasets
- [`docs/linkage.md`](docs/linkage.md) — linkage design
- [`docs/privacy_boundaries.md`](docs/privacy_boundaries.md) — privacy claim boundaries
- [`docs/privacy_backends.md`](docs/privacy_backends.md) — optional privacy/crypto research mechanisms
- [`docs/privacy_experiments.md`](docs/privacy_experiments.md) — empirical privacy research
- [`docs/threat_model.md`](docs/threat_model.md) — adversary and non-goals
- [`docs/reproducibility.md`](docs/reproducibility.md) — experiment reproducibility
- [`docs/independent_reproduction.md`](docs/independent_reproduction.md) — external reproduction procedure
- [`docs/limitations.md`](docs/limitations.md) — known limitations
- [`docs/release_policy.md`](docs/release_policy.md) — release policy
- [`paper/experiment_manifest.md`](paper/experiment_manifest.md) — paper experiment manifest

## Citation

If you use VertiMosaic in research or development, cite the software using [`CITATION.cff`](CITATION.cff). GitHub also exposes a **Cite this repository** action from that file.

```text
VertiMosaic
Saurav Singla
Version 0.2.0
Apache-2.0
https://github.com/sauravsingla/VertiMosaic
```

## License

VertiMosaic source code is licensed under the **Apache License 2.0**. Dataset licenses and provider terms remain separate; see [`DATA_LICENSES.md`](DATA_LICENSES.md).