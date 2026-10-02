---
title: VertiMosaic — Interactive VFL Evidence Explorer
emoji: 🧩
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: true
license: apache-2.0
short_description: Interactive VFL benchmark, robustness & provenance explorer
models:
- sauravsingla08/VertiMosaic-VFL-Reference-Models
datasets:
- sauravsingla08/VertiMosaic-VFL-Benchmark
tags:
- vertical-federated-learning
- federated-learning
- tabular
- tabular-classification
- model-benchmarking
- privacy-preserving-ml
- reproducible-research
---

# VertiMosaic — Interactive VFL Evidence Explorer

Explore reproducible **vertical federated learning (VFL)** evidence directly in the browser. The Space is generated from the same VertiMosaic experiment pipeline used to publish benchmark evidence, rather than from hand-entered demo values.

### What you can explore

- **Model comparison** — ROC-AUC, PR-AUC, F1, calibration, communication and systems measurements.
- **Party contribution** — synthetic Bank, Telecom, Insurance and Retail feature-view ablations.
- **Partial entity overlap** — intersection-only vs. availability-indicator approaches.
- **Party dropout** — inference-time dropout and training/inference availability scenarios.
- **Controlled feature drift** — mean, variance, missingness and categorical-frequency shifts.
- **Systems & provenance** — timing, memory, communication accounting, source commit, seed and payload schema.

Use the guided cards on the landing view to jump to a question, switch metrics interactively, download the generated `data.json`, or open the linked benchmark dataset for the broader evidence package.

## Reproducibility

The Space is static and browser-only. GitHub Actions regenerates `data.json` from VertiMosaic's experiment code, validates the project's claim boundaries, and publishes the resulting payload through Hugging Face Trusted Publishers.

- **Source:** [sauravsingla/VertiMosaic](https://github.com/sauravsingla/VertiMosaic)
- **Dataset:** [VertiMosaic-VFL-Benchmark](https://huggingface.co/datasets/sauravsingla08/VertiMosaic-VFL-Benchmark)
- **Models:** [VertiMosaic-VFL-Reference-Models](https://huggingface.co/sauravsingla08/VertiMosaic-VFL-Reference-Models)

## Privacy boundary

VertiMosaic demonstrates raw-feature locality in a research VFL simulator. It does **not** claim cryptographic privacy, private set intersection, MPC, homomorphic encryption, secure aggregation, collusion resistance, malicious-party security or formal differential privacy.

## Data boundary

The Space uses VertiMosaic-generated synthetic benchmark evidence. It does not redistribute UCI, OpenML or IEEE-CIS source rows. The separate four-industry external benchmark is semi-synthetic: its public source datasets do not represent the same real individuals.
