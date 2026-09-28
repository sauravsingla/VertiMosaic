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

Many organizations can describe the **same entities** while holding **different feature columns**. A bank may know payment behavior, a telecom provider may know service behavior, an insurer may know risk signals, and a retailer may know purchase behavior.

VertiMosaic lets researchers and ML engineers **build, benchmark, and audit vertical federated learning workflows while parties keep their raw feature tables local**.

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

<p align="center">
  <img src="assets/vertimosaic-architecture.svg" alt="VertiMosaic architecture" width="92%">
</p>

The active party owns the target. Other parties retain their raw feature matrices and return protocol-specific local outputs. The reference protocols exchange model- and training-related signals instead of pooling passive-party raw feature matrices at the active party.

> **Privacy boundary:** raw-feature locality is not the same as end-to-end cryptographic privacy. VertiMosaic documents the guarantees, assumptions, and non-guarantees of each research path explicitly.

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

If the project is useful, the best support is simple: **run it, reproduce it, and share what happened**.

> **Note:** the measured benchmark below is a separate **800-row formal release run**; it is not the output of the `--rows 2000` quick-start demo above.

## Measured v0.3.0 benchmark snapshot

The table below is copied from the **machine-generated v0.3.0 formal benchmark evidence** for the core reference run (**800 rows, seed 42**). The first two rows are explicitly **non-federated baselines**; they are not VFL protocols. Displayed values are rounded to four decimals for readability; the linked release artifacts retain full precision.

| Model / protocol | Setting | ROC-AUC | PR-AUC | F1 | Train time (s) |
|---|---|---:|---:|---:|---:|
| Centralized all features | Non-federated baseline | 0.7786 | 0.7769 | 0.7206 | 0.0037 |
| Single-party Bank | Non-federated baseline | 0.6566 | 0.6241 | 0.6080 | 0.0030 |
| `VFLLogisticRegression` | Federated | 0.7872 | 0.7860 | 0.7361 | 0.0222 |
| `VFLHistGBDT` | Federated | 0.7041 | 0.6749 | 0.6622 | 0.0631 |

These values are a **single reproducible reference run, not evidence that VFL generally outperforms centralized training and not a cross-framework leaderboard**. Wall-clock time is environment-dependent. The full generated comparison also records inference time, sampled process RSS, protocol payload bytes/message counts, Brier score, and a simple empirical membership-attack baseline. Protocol communication values are application-level payload accounting, not packet captures.

### Inspect the release evidence

| Evidence | What it contains |
|---|---|
| [`formal_comparison.md`](https://github.com/sauravsingla/VertiMosaic/releases/download/v0.3.0/formal_comparison.md) | Full human-readable formal comparison |
| [`canonical-results.json`](https://github.com/sauravsingla/VertiMosaic/releases/download/v0.3.0/canonical-results.json) | Canonical release-result summary |
| [`remote_transport.json`](https://github.com/sauravsingla/VertiMosaic/releases/download/v0.3.0/remote_transport.json) | Serialized remote-transport measurements |
| [Full v0.3.0 release](https://github.com/sauravsingla/VertiMosaic/releases/tag/v0.3.0) | Manifest, hashes, privacy audits, environment capture, preprint, and packaged evidence |

Maintainer-generated release evidence demonstrates reproducibility of the maintained pipeline; it is **not the same as independent third-party reproduction**. Independent reproduction is supported under [`reproductions/`](reproductions/) and [`docs/independent_reproduction.md`](docs/independent_reproduction.md).

## Why VertiMosaic?

- **CPU-first tabular VFL** — reference logistic and histogram-GBDT protocols run without dedicated accelerators.
- **Protocol correctness** — strict entity alignment can reject equal-length but differently ordered party rows.
- **Auditable experiments** — communication, runtime, memory, metrics, provenance, and reproducibility metadata are recorded explicitly.
- **Linked public benchmarks** — includes exact-NPI multi-source provider linkage and exact multi-table MovieLens linkage.
- **Explicit privacy boundaries** — raw-feature locality is kept separate from stronger DP, PSI, MPC, HE, secure-aggregation, or malicious-party guarantees.

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

## Core scope

| Property | Current scope |
|---|---|
| Federation | Vertical federated learning |
| Models | `VFLLogisticRegression`, `VFLHistGBDT` |
| Compute | CPU supported; GPU not required |
| Raw passive feature tables pooled during VFL | No |
| Entity alignment | Order-sensitive validation in strict research paths |
| Public linked benchmarks | NPI multi-source, MovieLens multi-table, UCI exact-row; optional local IEEE-CIS |
| Reproducibility | Run bundles, release evidence, hashes, environment capture, reproduction schemas |

## Technical depth

The README summarizes the research surface; detailed protocol, privacy, benchmark, and reproducibility definitions live in the linked documentation.

### Protocols and transport

`VFLLogisticRegression` is a first-principles NumPy reference protocol. `VFLHistGBDT` is a vertical histogram-gradient-boosting reference implementation in which parties retain local training-derived bins and return aggregate histogram candidates plus opaque feature/bin references.

`InMemoryTransport` supports deterministic simulation and message auditing. `RemoteHTTPTransport` adds a serialized HTTP path with bounded requests/responses, authorization, replay controls, rate limiting, optional compressed NumPy transport, and signed-message support.

Details: [`docs/architecture.md`](docs/architecture.md) · [`docs/protocol.md`](docs/protocol.md)

### Alignment and protected path

Strict research paths bind ordered pseudonymous entity identifiers to party partitions and derive order-sensitive digests, so same-length inputs can still be rejected when entity identities or ordering differ. Missing trained parties fail by default; the research-only `zero_contribution` fallback must be selected explicitly.

The protected logistic workflow can combine configured PSI-based intersection, canonical entity ordering, strict alignment metadata, and clipped-Gaussian protection for active-to-passive residual releases. Its guarantee is narrow: **logits, model parameters, timing, routing information, other protocol messages, and transport metadata remain outside that composite guarantee.**

Details: [`docs/protected_logistic.md`](docs/protected_logistic.md) · [`docs/privacy_boundaries.md`](docs/privacy_boundaries.md) · [`docs/threat_model.md`](docs/threat_model.md)

### Benchmarks and evaluation

| Benchmark | Linkage category |
|---|---|
| NPI provider benchmark | Real exact multi-source public linkage |
| MovieLens 1M | Public exact multi-table linkage |
| IEEE-CIS | Authorized local exact linkage |
| UCI Credit | Exact-row vertical partition |
| Bank / Telecom / Insurance / Retail | Explicitly semi-synthetic cross-domain linkage |
| Synthetic scale / overlap | Controlled synthetic study |

The four-industry benchmark does **not** claim that its public source datasets describe the same real individuals. Entity-level splits default to **70% train / 15% validation / 15% test**, with threshold selection based on validation predictions only. Standard run bundles capture configuration, provenance, environment versions, predictions, metrics, communication metadata, hashes, timestamps, seed, and Git state.

Details: [`docs/linked_benchmarks.md`](docs/linked_benchmarks.md) · [`docs/formal_benchmark.md`](docs/formal_benchmark.md) · [`docs/release_evidence.md`](docs/release_evidence.md)

### Comparators and privacy boundary

VertiMosaic includes a deterministic export/import contract for measured comparisons with frameworks such as **FATE** and **SecretFlow**. A wrapper or exchange format is **not** treated as a measured third-party result; comparator rows remain pending until those frameworks are actually run against the frozen exchange bundle.

The default VFL protocols provide raw-feature locality, party-local computation, explicit message boundaries, and auditable research transports. They do **not** automatically provide end-to-end differential privacy, PSI, MPC, homomorphic encryption, secure aggregation, collusion resistance, or malicious-party security. Optional research mechanisms exist, but none should be interpreted as a blanket **"secure VFL"** guarantee.

Details: [`docs/external_comparators.md`](docs/external_comparators.md) · [`docs/privacy_backends.md`](docs/privacy_backends.md) · [`docs/privacy_boundaries.md`](docs/privacy_boundaries.md) · [`docs/threat_model.md`](docs/threat_model.md)

## What is new in v0.3.0

- order-sensitive entity alignment and explicit missing-party behavior;
- protected logistic research path with PSI-based alignment and scoped clipped-Gaussian residual releases;
- exact-NPI multi-source public benchmark builder;
- external-comparator exchange contract for measured FATE/SecretFlow comparisons;
- independent-reproduction evidence schemas and validation tooling; and
- immutable release-evidence assets, focused tutorials, and a separate-service HTTPS/mTLS deployment example.

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

## Documentation

The working technical report is [`paper/preprint.md`](paper/preprint.md), with the experiment plan in [`paper/experiment_manifest.md`](paper/experiment_manifest.md).

Key references:

- [`docs/architecture.md`](docs/architecture.md) — architecture and component boundaries
- [`docs/protocol.md`](docs/protocol.md) — protocol details and alignment behavior
- [`docs/formal_benchmark.md`](docs/formal_benchmark.md) — benchmark matrix and interpretation boundaries
- [`docs/linked_benchmarks.md`](docs/linked_benchmarks.md) — linked-benchmark taxonomy
- [`docs/external_comparators.md`](docs/external_comparators.md) — comparator exchange and measurement contract
- [`docs/privacy_boundaries.md`](docs/privacy_boundaries.md) — privacy claim boundaries
- [`docs/threat_model.md`](docs/threat_model.md) — adversary and non-goals
- [`docs/independent_reproduction.md`](docs/independent_reproduction.md) — external reproduction procedure
- [`docs/release_evidence.md`](docs/release_evidence.md) — immutable release evidence
- [`docs/limitations.md`](docs/limitations.md) — known limitations and non-claims

## Reproduce or contribute

The most valuable external signal for VertiMosaic is an **independent run**.

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