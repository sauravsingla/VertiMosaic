<h1 align="center">VertiMosaic</h1>

<p align="center">
  <b>Train one model across organizations without pooling their raw tabular features — on CPU.</b>
</p>

<p align="center">
  CPU-first vertical federated learning for aligned entities, heterogeneous feature sets, reproducible experiments, and explicit privacy boundaries.
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

## The problem VertiMosaic solves

Many organizations can describe the **same entities** but hold **different feature columns**.

A bank may know payment behavior. A telecom provider may know service behavior. An insurer may know claims or risk signals. A retailer may know purchase behavior. Pooling all of those raw feature tables into one place may be undesirable, restricted, or simply unrealistic.

**VertiMosaic is a research framework for studying how those parties can participate in one vertical federated learning workflow while keeping passive-party raw feature matrices local.**

```text
                same / aligned entities
                         │
       ┌─────────────────┼─────────────────┐
       │                 │                 │
     Bank             Telecom          Insurance          Retail
 features + target     features          features          features
       │                 │                 │                 │
       └──────── party-local computation + protocol messages ────────┘
                         │
                   joint VFL model
```

The reference protocols exchange model- and training-related signals rather than pooling passive-party raw feature tables at the active party. That is a useful system property, but **it is not the same as a blanket cryptographic or end-to-end privacy guarantee**; VertiMosaic keeps those boundaries explicit.

## Try it in under a minute

VertiMosaic supports Python **3.11 and 3.12** and does not require a GPU.

```bash
python -m pip install vertimosaic
vertimosaic demo --rows 2000 --seed 42
```

The demo runs a CPU-friendly smoke experiment and writes a reproducibility bundle under `runs/<run_id>/`.

Want the code instead of the package?

```bash
git clone https://github.com/sauravsingla/VertiMosaic.git
cd VertiMosaic
python -m pip install -e .
vertimosaic demo --rows 2000 --seed 42
```

## Why VertiMosaic?

- **CPU-first** — useful for laptops, CI, GitHub-hosted environments, and reproducible research without dedicated accelerators.
- **Built for tabular VFL** — reference logistic and histogram-GBDT protocols rather than a generic distributed-training wrapper.
- **Strict entity alignment** — research paths can reject equal-length but differently ordered party rows instead of silently treating them as aligned.
- **Auditable communication** — protocol messages, payload accounting, and separate serialized HTTP benchmark paths are measurable.
- **Real linked public benchmarks** — including exact NPI linkage across multiple US public provider data products and exact multi-table MovieLens linkage.
- **Explicit privacy boundaries** — raw-feature locality is separated from stronger claims about DP, PSI, MPC, HE, secure aggregation, or malicious-party security.
- **Reproducibility first** — deterministic runs, release evidence, environment capture, hashes, benchmark outputs, and independent-reproduction schemas.

## Pick your path

| If you want to... | Start here |
|---|---|
| Understand VFL quickly | [`tutorials/01_first_vfl_10_minutes.md`](tutorials/01_first_vfl_10_minutes.md) |
| Run the basic demo | `vertimosaic demo --rows 2000 --seed 42` |
| Compare centralized / single-party / VFL paths | `vertimosaic-benchmark-matrix` |
| Run a public exact-linked benchmark | `vertimosaic-linked-movielens` |
| Inspect privacy limitations before using the framework | [`docs/privacy_boundaries.md`](docs/privacy_boundaries.md) |
| Reproduce the maintained evidence bundle | `vertimosaic-reproduce` |
| Try separate services with HTTPS/mTLS | [`deploy/compose/README.md`](deploy/compose/README.md) |
| Explore the live project surface | [Hugging Face Space](https://huggingface.co/spaces/sauravsingla08/VertiMosaic) |

## What VertiMosaic is — and is not

| VertiMosaic provides | VertiMosaic does not automatically claim |
|---|---|
| Party-local raw feature tables in the reference VFL paths | End-to-end cryptographic privacy |
| Explicit protocol/message boundaries | Protection of every derived signal |
| CPU-friendly reference implementations | Production-scale distributed infrastructure |
| Optional PSI / DP / secure-sum / HE research mechanisms | A blanket "secure VFL" guarantee |
| Reproducible public and synthetic benchmarks | Evidence of real private cross-company collaboration |

## At a glance

| Property | Current scope |
|---|---|
| Federation | Vertical federated learning |
| Reference models | `VFLLogisticRegression`, `VFLHistGBDT` |
| Protocol alignment | Order-sensitive entity IDs/digests with exact-match validation in strict research paths |
| Missing-party default | Error; zero-contribution fallback is explicit opt-in research behavior |
| Active party in the four-industry benchmark | Bank, with target |
| Passive parties | Telecom, Insurance, Retail |
| Raw passive feature tables pooled during VFL | No |
| Compute | CPU supported; GPU not required |
| Real exact multi-source public linkage | NPI-linked Open Payments + NPPES + CMS Care Compare/provider data |
| Public exact-linked benchmark | MovieLens 1M multi-table linkage |
| Additional linked benchmarks | UCI exact-row vertical partition; authorized local IEEE-CIS transaction/identity linkage |
| Cross-industry benchmark | Public-source, explicitly semi-synthetic linkage |
| Protected logistic research path | PSI alignment + clipped-Gaussian residual releases with scoped accounting |
| Default privacy posture | Raw-feature locality; not end-to-end cryptographic VFL |
| Optional privacy research mechanisms | Clipped Gaussian/zCDP releases, secure-sum, additive sharing, PSI, Paillier |
| External framework comparison | Deterministic exchange/normalization contract available; measured third-party runs remain external work |

## Architecture

<p align="center">
  <img src="assets/vertimosaic-architecture.svg" alt="VertiMosaic architecture" width="100%">
</p>

The active party owns the target. Passive parties retain their raw feature matrices and return only protocol-specific local outputs. `VFLLogisticRegression` exchanges local logits/residual-related signals for party-local gradient updates. `VFLHistGBDT` exchanges target-derived gradient/Hessian signals and aggregate histogram candidate statistics while numeric split thresholds and routing state remain with the owning party.

`InMemoryTransport` provides deterministic protocol simulation and message auditing. `RemoteHTTPTransport` provides a separate serialized HTTP path with bounded requests/responses, authorization, replay controls, idempotency limits, rate limiting, optional compressed NumPy transport, and signed-message support. The Docker Compose deployment example adds separate services, party-local volumes, HTTPS/mTLS, bearer authorization, health checks, and deterministic sample data. These transport controls do not turn the default learning protocol into malicious-secure VFL.

## What is new in v0.3.0

VertiMosaic v0.3.0 strengthens the research and protocol surface with:

- **order-sensitive entity alignment**;
- **explicit missing-party behavior** in `VFLLogisticRegression`;
- a **protected logistic research path** combining PSI-based entity intersection/canonical ordering with clipped-Gaussian residual releases and scoped zCDP accounting;
- a **real exact-NPI multi-source public benchmark builder** linking provider entities across Open Payments, NPPES, and CMS Care Compare/provider data;
- a framework-neutral **external-comparator exchange contract** for FATE, SecretFlow, or another VFL framework, without inventing unmeasured third-party results;
- **independent-reproduction evidence schemas and validation tooling**;
- four focused tutorials and a separate-service **Docker Compose HTTPS/mTLS deployment example**; and
- immutable release-evidence assets containing canonical results, benchmark outputs, privacy audits, environment metadata, hashes, and reproduction artifacts.

Latest release: **[VertiMosaic v0.3.0](https://github.com/sauravsingla/VertiMosaic/releases/tag/v0.3.0)**.

## Installation and tutorials

For development and verification tooling:

```bash
python -m pip install -e ".[dev]"
```

Optional PSI and Paillier adapters:

```bash
python -m pip install -e ".[privacy-crypto]"
```

Start with the focused walkthroughs:

- [`tutorials/01_first_vfl_10_minutes.md`](tutorials/01_first_vfl_10_minutes.md) — first VFL run in about 10 minutes
- [`tutorials/02_two_party_logistic.md`](tutorials/02_two_party_logistic.md) — strict two-party logistic VFL
- [`tutorials/03_histogram_gbdt.md`](tutorials/03_histogram_gbdt.md) — vertical histogram GBDT
- [`tutorials/04_privacy_leakage_and_mitigation.md`](tutorials/04_privacy_leakage_and_mitigation.md) — leakage, audits, and scoped mitigations

For separate-service execution with HTTPS/mTLS, see [`deploy/compose/README.md`](deploy/compose/README.md).

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

## Protocol correctness

### Entity alignment

Strict research paths bind ordered pseudonymous entity identifiers to party partitions and derive order-sensitive digests. Training, validation, and inference can therefore reject party collections whose entity sets or ordering do not match exactly.

This is intentionally stronger than checking only row counts: two parties can have the same number of rows while referring to different entities or different row orders.

See [`docs/protocol.md`](docs/protocol.md) and the alignment implementation under [`src/vertimosaic/alignment/`](src/vertimosaic/alignment/).

### Missing parties

`VFLLogisticRegression` records the parties present during training. At inference, omission of a trained party is an error by default. The research-only `zero_contribution` fallback must be selected explicitly; it is not a silent default.

This keeps dropout experiments possible without allowing accidental party omission to pass unnoticed in ordinary inference.

## Reference protocols

### `VFLLogisticRegression`

A first-principles NumPy reference protocol. Each party computes logits and gradients using only its local feature matrix. Passive logits and residual-related signals pass through the explicit message/transport abstraction.

### `VFLHistGBDT`

A vertical histogram-gradient-boosting reference implementation. Each party fits and retains its own training-derived quantile bins. Passive parties compute aggregate histogram candidates from received gradient/Hessian signals and return aggregate statistics plus opaque feature/bin references. The active party selects a split; the owning party resolves the private numeric threshold and performs routing locally. Validation and test rows reuse training-derived routing state rather than refitting held-out bins.

Centralized models are provided only as **non-federated baselines**. They deliberately pool selected feature columns and must not be described as VFL.

## Protected logistic research path

VertiMosaic v0.3.0 includes an explicitly scoped protected logistic workflow in [`src/vertimosaic/privacy/protected_logistic.py`](src/vertimosaic/privacy/protected_logistic.py).

The path can:

1. intersect party entity sets through a configured PSI backend;
2. canonicalize all parties to one exact ordered entity sequence;
3. bind strict entity metadata to the aligned partitions; and
4. train `VFLLogisticRegression` with `ClippedGaussianDPBackend` applied to active-to-passive residual releases.

The resulting privacy report is deliberately narrow. PSI protects set intersection according to the selected PSI backend's threat model, while clipped-Gaussian accounting covers the protected residual-release family only. **Logits, model parameters, timing, other protocol messages, routing information, and transport metadata remain outside that composite guarantee.**

See [`docs/protected_logistic.md`](docs/protected_logistic.md), [`docs/privacy_boundaries.md`](docs/privacy_boundaries.md), and [`docs/threat_model.md`](docs/threat_model.md).

## Benchmarks and linkage categories

VertiMosaic keeps linkage categories explicit so reproducibility is not confused with stronger real-world linkage evidence.

### 1. NPI-linked public provider benchmark — real exact multi-source linkage

The v0.3 benchmark builder in [`src/vertimosaic/datasets/npi_public.py`](src/vertimosaic/datasets/npi_public.py) links real provider entities using the authoritative 10-digit **National Provider Identifier (NPI)** across distinct public US government data products.

The active side can use prior-period Open Payments aggregates, the target can be derived from a later Open Payments period, and passive features can come from NPPES and CMS Care Compare/provider data for the exact same ordered provider entities.

This is genuine exact **multi-source public linkage** over real providers. It must **not** be described as evidence of private cross-company collaboration, private data sharing, or production PSI. See [`docs/npi_linked_benchmark.md`](docs/npi_linked_benchmark.md).

### 2. MovieLens 1M — public exact multi-table linkage

`vertimosaic-linked-movielens` downloads MovieLens 1M directly from GroupLens and links its observed `users.dat`, `ratings.dat`, and `movies.dat` identifiers. The active side uses user demographics; the passive side derives historical rating/genre behavior from the earlier portion of each user's timeline; the target is derived from later rating events that are excluded from passive features.

This is **genuine exact linkage across public tables describing the same service users**. It is still a single-service dataset, not cross-organization entity resolution and not evidence of PSI.

### 3. IEEE-CIS — authorized local exact linkage

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

### 4. UCI Credit — exact-row vertical partition

`vertimosaic-linked-uci` retrieves UCI Default of Credit Card Clients at runtime and partitions disjoint source columns between active and passive parties while preserving exact entity rows. It is a useful public linkage sanity benchmark, but the two partitions originate from one source dataset and therefore do not demonstrate cross-organization entity resolution.

### 5. Four-industry benchmark — semi-synthetic cross-domain linkage

The Bank/Telecom/Insurance/Retail benchmark uses real public source-domain data but the four sources do **not** describe the same real individuals. They are independently preprocessed and connected by an explicitly documented research linkage mechanism.

| Party | Public source | Role |
|---|---|---|
| Bank | UCI Default of Credit Card Clients (350) | Financial/payment behavior + observed target |
| Telecom | UCI Iranian Churn (563) | Telecom/service behavior |
| Insurance | OpenML `freMTPL2freq` + `freMTPL2sev` (41214/41215) | Claims/risk behavior |
| Retail | UCI Online Retail (352) | Purchase behavior aggregated to customer level |

`observed_target_external` uses the Bank target with target-blind external linkage. `distributed_signal_external` links transformed source-domain features first and generates the disclosed semi-synthetic target only after the linked profiles are formed.

### 6. Controlled synthetic and overlap studies

`synthetic_scale` supports controlled scaling, overlap, dropout, drift, noise, and distributed-signal experiments. `vertimosaic overlap` evaluates **100%, 90%, 75%, 50%, and 25%** Bank-to-passive overlap while preserving deterministic entity splits and reporting coverage, utility, and communication separately for each strategy.

For full linkage definitions and limitations, see [`docs/linked_benchmarks.md`](docs/linked_benchmarks.md), [`docs/npi_linked_benchmark.md`](docs/npi_linked_benchmark.md), and [`docs/linkage.md`](docs/linkage.md).

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

## External VFL comparators

VertiMosaic provides a deterministic export/import contract for comparison with established VFL frameworks such as **FATE** or **SecretFlow**:

- [`scripts/export_comparator_benchmark.py`](scripts/export_comparator_benchmark.py) freezes aligned party arrays, targets, entity IDs, splits, and SHA-256 metadata;
- [`scripts/run_external_comparator.py`](scripts/run_external_comparator.py) normalizes measured external-framework output into the repository's comparison schema;
- [`docs/external_comparators.md`](docs/external_comparators.md) documents the contract and claim boundaries.

The repository does **not** treat a wrapper or exchange format as if it were a measured FATE/SecretFlow result. Third-party comparator rows remain pending until those frameworks are actually run and measured against the frozen exchange bundle.

## Release evidence

The **v0.3.0 GitHub release** includes machine-generated evidence assets rather than manually entered headline numbers. The release-evidence workflow reproduces the immutable release source and publishes, among other files:

- `canonical-results.json`;
- `manifest.json` with hashed evidence references;
- formal comparison CSV/JSON/Markdown outputs;
- serialized remote-transport measurements;
- empirical and advanced privacy-audit results;
- multi-seed privacy-research results;
- exact `pip-freeze.txt` environment capture;
- the technical preprint; and
- a packaged `vertimosaic-release-evidence.tar.gz` plus SHA-256 checksum.

See the **[v0.3.0 release assets](https://github.com/sauravsingla/VertiMosaic/releases/tag/v0.3.0)** and [`docs/release_evidence.md`](docs/release_evidence.md).

Maintainer CI and release evidence demonstrate reproducibility of the maintained pipeline, but they are **not the same as independent third-party reproduction**. The repository provides a separate attestation schema and validation path under [`reproductions/`](reproductions/) and [`docs/independent_reproduction.md`](docs/independent_reproduction.md).

## Privacy and security boundary

The default VFL protocols provide raw-feature locality, party-local preprocessing/computation, explicit message boundaries, and auditable research transports. They do **not** automatically provide end-to-end differential privacy, PSI, MPC, homomorphic encryption, secure aggregation, collusion resistance, or malicious-party security.

Sensitive derived signals—including gradients, Hessians, logits, residuals, entity membership, routing information, and split statistics—may leak information depending on the observer and protocol.

VertiMosaic v0.3.0 includes **optional, explicitly scoped research mechanisms**:

- `ClippedGaussianDPBackend` for message-level L2 clipping, Gaussian noise, zCDP composition, and `(epsilon, delta)` conversion;
- `PairwiseMaskSecureAggregation` as an honest-but-curious secure-sum reference primitive;
- `AdditiveSecretSharingSum` for additive-share sum experiments;
- `OpenMinedPSIBackend` as an optional PSI adapter; and
- `PaillierHomomorphicSum` for bounded additive homomorphic-sum experiments.

The protected logistic path composes some of these mechanisms for one explicit research workflow, but none of the primitives should be interpreted as a blanket **"secure VFL"** guarantee. See [`docs/privacy_backends.md`](docs/privacy_backends.md), [`docs/privacy_experiments.md`](docs/privacy_experiments.md), [`docs/privacy_boundaries.md`](docs/privacy_boundaries.md), and [`docs/threat_model.md`](docs/threat_model.md).

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

CI exercises Python 3.11/3.12, protocol-critical coverage, paper/reproduction smoke paths, privacy audits, remote transport measurement, package validation, CodeQL, dependency audit/SBOM generation, optional crypto backends, clean-room reproduction, external-data integration, and portability smoke checks on Linux ARM64, macOS, and Windows. Protocol-critical modules target at least **90% coverage**.

## Research report and documentation

The working technical report is [`paper/preprint.md`](paper/preprint.md), with the full experiment plan in [`paper/experiment_manifest.md`](paper/experiment_manifest.md).

Key documentation:

- [`docs/architecture.md`](docs/architecture.md) — architecture and component boundaries
- [`docs/protocol.md`](docs/protocol.md) — reference protocol details and alignment behavior
- [`docs/formal_benchmark.md`](docs/formal_benchmark.md) — benchmark matrix and interpretation boundaries
- [`docs/linked_benchmarks.md`](docs/linked_benchmarks.md) — exact-linked and semi-synthetic benchmark taxonomy
- [`docs/npi_linked_benchmark.md`](docs/npi_linked_benchmark.md) — exact-NPI multi-source public benchmark
- [`docs/external_comparators.md`](docs/external_comparators.md) — FATE/SecretFlow exchange and measurement contract
- [`docs/protected_logistic.md`](docs/protected_logistic.md) — PSI + clipped-Gaussian protected logistic research path
- [`docs/privacy_boundaries.md`](docs/privacy_boundaries.md) — privacy claim boundaries
- [`docs/privacy_backends.md`](docs/privacy_backends.md) — optional privacy/crypto research mechanisms
- [`docs/privacy_experiments.md`](docs/privacy_experiments.md) — empirical privacy research
- [`docs/threat_model.md`](docs/threat_model.md) — adversary and non-goals
- [`docs/reproducibility.md`](docs/reproducibility.md) — experiment reproducibility
- [`docs/independent_reproduction.md`](docs/independent_reproduction.md) — external reproduction procedure
- [`docs/release_evidence.md`](docs/release_evidence.md) — immutable release evidence
- [`docs/limitations.md`](docs/limitations.md) — known limitations and non-claims
- [`docs/release_policy.md`](docs/release_policy.md) — release policy

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
