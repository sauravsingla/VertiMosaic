# Quick Start

This path gets from a clean Python environment to a first VertiMosaic run without requiring a GPU or private data.

## Requirements

- Python **3.11 or 3.12**
- a CPU environment
- `pip`

## Install the released package

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install vertimosaic
```

Confirm the CLI is available:

```bash
vertimosaic --help
```

## Run the first deterministic demo

```bash
vertimosaic demo --rows 2000 --seed 42
```

The demo is a smoke experiment intended to verify installation and the basic VFL execution path. Research benchmark paths use stronger evidence generation and larger bootstrap settings where practical.

## Run the formal comparison

```bash
vertimosaic-benchmark-matrix
```

This is the main centralized / single-party / VFL comparison entry point. Use the generated artifacts rather than manually transcribing metrics when reporting results.

## Create an independent reproduction bundle

For a released version such as `0.3.0`:

```bash
python -m pip install vertimosaic==0.3.0
vertimosaic-reproduce --expected-version 0.3.0 --output evidence
```

The evidence directory is designed for inspection and independent sharing. See [Independent Reproduction](independent_reproduction.md) for the expected artifacts and attestation model.

## Optional privacy research dependencies

PSI and Paillier adapters are intentionally optional:

```bash
python -m pip install "vertimosaic[privacy-crypto]"
```

Installing these dependencies does **not** make every VertiMosaic protocol cryptographically private. Read [Privacy Boundaries](privacy_boundaries.md), [Threat Model](threat_model.md), and [Research Backends](privacy_backends.md) before describing the guarantee of an experiment.

## Work from the repository

```bash
git clone https://github.com/sauravsingla/VertiMosaic.git
cd VertiMosaic
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the verification suite:

```bash
make coverage
make coverage-protocol
```

The CI also checks formatting, linting, static typing, package build integrity, protocol-critical coverage, reproducibility smoke paths, privacy audits, and serialized transport behavior.

## Choose the next path

| Goal | Next step |
|---|---|
| Understand party/model boundaries | [Architecture](architecture.md) |
| Understand message flow and alignment | [Protocol](protocol.md) |
| Evaluate privacy guarantees | [Privacy Boundaries](privacy_boundaries.md) |
| Select an appropriate dataset | [Benchmarks](benchmarks.md) |
| Reproduce a release | [Reproducibility](reproducibility.md) |
| Use public classes directly | [API Reference](api.md) |

## Focused tutorials

The repository also contains four hands-on tutorials:

1. [First VFL in 10 minutes](https://github.com/sauravsingla/VertiMosaic/blob/main/tutorials/01_first_vfl_10_minutes.md)
2. [Two-party logistic VFL](https://github.com/sauravsingla/VertiMosaic/blob/main/tutorials/02_two_party_logistic.md)
3. [Histogram GBDT](https://github.com/sauravsingla/VertiMosaic/blob/main/tutorials/03_histogram_gbdt.md)
4. [Privacy leakage and mitigation](https://github.com/sauravsingla/VertiMosaic/blob/main/tutorials/04_privacy_leakage_and_mitigation.md)

For a separate-service developer-laptop example, see the [Docker Compose HTTPS/mTLS deployment](https://github.com/sauravsingla/VertiMosaic/tree/main/deploy/compose).
